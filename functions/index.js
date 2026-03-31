const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { GoogleGenerativeAI } = require("@google/generative-ai");

admin.initializeApp();
const db = admin.firestore();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || functions.config().gemini?.api_key;
const DASHBOARD_SECRET = process.env.DASHBOARD_SECRET || functions.config().dashboard?.secret;
const GEMINI_MODEL = "gemini-3.1-flash-lite-preview";
const BATCH_SIZE = 20;          // conversations per Claude call
const MAX_MESSAGES_PER_CHAT = 10; // keep tokens low per conversation
const MAX_BATCHES_PER_RUN = 5;    // max 100 conversations per button click
const DELAY_BETWEEN_BATCHES = 5000; // 5 seconds between batches

exports.analyzeChats = functions
  .runWith({ timeoutSeconds: 540, memory: "1GB" })
  .https.onRequest(async (req, res) => {
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") return res.status(204).send("");

    // Auth check
    const secret = req.query.secret;
    if (!secret || secret !== DASHBOARD_SECRET) {
      return res.status(401).json({ success: false, error: "Unauthorized" });
    }

    const action = req.query.action;
    const days = parseInt(req.query.days) || 30;

    // Ping: just verify auth is valid
    if (action === "ping") {
      return res.json({ success: true });
    }

    // Conversations tab: return raw conversations for browsing
    if (action === "conversations") {
      try {
        const since = new Date();
        since.setDate(since.getDate() - days);
        const conversations = await fetchConversationsRaw(since);
        return res.json({ success: true, conversations });
      } catch (error) {
        console.error("Conversations fetch failed:", error);
        return res.status(500).json({ success: false, error: error.message });
      }
    }

    // Default: run analysis
    try {
      const fullRun = req.query.full === "true";

      // Load already-analyzed session IDs from Firestore
      const metaDoc = await db.doc("chatInsights/_metadata").get();
      const meta = metaDoc.exists ? metaDoc.data() : {};
      const analyzedIds = new Set(fullRun ? [] : (meta.analyzedSessionIds || []));

      console.log(`Starting analysis: days=${days}, fullRun=${fullRun}, alreadyAnalyzed=${analyzedIds.size}`);

      const since = new Date();
      since.setDate(since.getDate() - days);

      const { conversations, newSessionIds } = await fetchConversations(since, analyzedIds);
      console.log(`Found ${conversations.length} new conversations to analyze`);

      if (conversations.length === 0) {
        return res.json({ status: "ok", message: "No new conversations to analyze", totalNew: 0 });
      }

      // Analyze with Gemini in batches (capped per run to avoid rate limits)
      const client = new GoogleGenerativeAI(GEMINI_API_KEY);
      const batchResults = [];
      const conversationsThisRun = conversations.slice(0, MAX_BATCHES_PER_RUN * BATCH_SIZE);
      const remaining = conversations.length - conversationsThisRun.length;

      for (let i = 0; i < conversationsThisRun.length; i += BATCH_SIZE) {
        const batch = conversationsThisRun.slice(i, i + BATCH_SIZE);
        const batchNum = Math.floor(i / BATCH_SIZE) + 1;
        console.log(`Analyzing batch ${batchNum} of ${Math.ceil(conversationsThisRun.length / BATCH_SIZE)} (${batch.length} conversations)`);
        const result = await analyzeWithClaude(client, batch);
        batchResults.push(result);
        // Pause between batches to stay within rate limits
        if (i + BATCH_SIZE < conversationsThisRun.length) {
          await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES));
        }
      }

      // Only mark the conversations we actually processed as analyzed
      const processedIds = newSessionIds.slice(0, conversationsThisRun.length);

      const mergedResult = mergeBatchResults(batchResults, since, conversationsThisRun.length);
      await saveInsights(mergedResult, processedIds);

      console.log(`Analysis complete: ${conversationsThisRun.length} analyzed, ${remaining} remaining`);
      return res.json({
        status: "ok",
        result: mergedResult,
        totalNew: conversationsThisRun.length,
        remaining,
        message: remaining > 0 ? `Analyzed ${conversationsThisRun.length} conversations. ${remaining} more remain — click again to continue.` : `Analyzed ${conversationsThisRun.length} conversations. All caught up!`
      });

    } catch (error) {
      console.error("Analysis failed:", error);
      return res.status(500).json({ status: "error", message: error.message });
    }
  });

/**
 * Fetch conversations not yet analyzed.
 * Looks back `days` days, skips sessions already in analyzedIds.
 */
async function fetchConversations(since, analyzedIds) {
  const conversations = [];
  const newSessionIds = [];
  const usersSnap = await db.collection("users").get();

  for (const userDoc of usersSnap.docs) {
    const sessionsSnap = await userDoc.ref
      .collection("chatSessions")
      .where("createdAt", ">=", since)
      .orderBy("createdAt", "desc")
      .get();

    for (const sessionDoc of sessionsSnap.docs) {
      const sessionId = sessionDoc.id;

      // Skip if already analyzed
      if (analyzedIds.has(sessionId)) continue;

      const sessionData = sessionDoc.data();
      const messagesSnap = await sessionDoc.ref
        .collection("messages")
        .orderBy(admin.firestore.FieldPath.documentId())
        .limit(MAX_MESSAGES_PER_CHAT)
        .get();

      const messages = [];
      for (const msgDoc of messagesSnap.docs) {
        const msgData = msgDoc.data();
        const docId = msgDoc.id;
        let role = "user";
        if (docId.startsWith("developer_")) role = "assistant";
        else if (docId.startsWith("article_")) role = "system";
        else if (msgData.role === "assistant" || msgData.role === "developer") role = "assistant";
        else if (msgData.role === "system") role = "system";

        if (msgData.content && msgData.content.trim()) {
          messages.push({ role, content: msgData.content.trim() });
        }
      }

      // Only include real conversations (at least 2 messages — a user reply exists)
      if (messages.length >= 2) {
        conversations.push({
          sessionId,
          userId: userDoc.id,
          createdAt: sessionData.createdAt?.toDate?.() || null,
          messageCount: sessionData.messageCount || messages.length,
          chatTitle: sessionData.chatTitle || null,
          messages,
        });
        newSessionIds.push(sessionId);
      }
    }
  }

  return { conversations, newSessionIds };
}

/**
 * Fetch raw conversations for the Conversations browser tab.
 * No filtering by analyzed state — shows everything in the date range.
 */
async function fetchConversationsRaw(since) {
  const conversations = [];
  const usersSnap = await db.collection("users").get();

  for (const userDoc of usersSnap.docs) {
    const sessionsSnap = await userDoc.ref
      .collection("chatSessions")
      .where("createdAt", ">=", since)
      .orderBy("createdAt", "desc")
      .get();

    for (const sessionDoc of sessionsSnap.docs) {
      const sessionData = sessionDoc.data();
      const messagesSnap = await sessionDoc.ref
        .collection("messages")
        .orderBy(admin.firestore.FieldPath.documentId())
        .limit(MAX_MESSAGES_PER_CHAT)
        .get();

      const messages = [];
      for (const msgDoc of messagesSnap.docs) {
        const msgData = msgDoc.data();
        const docId = msgDoc.id;
        let role = "user";
        if (docId.startsWith("developer_")) role = "assistant";
        else if (docId.startsWith("article_")) role = "system";
        else if (msgData.role === "assistant" || msgData.role === "developer") role = "assistant";
        else if (msgData.role === "system") role = "system";

        if (msgData.content && msgData.content.trim()) {
          messages.push({ role, content: msgData.content.trim() });
        }
      }

      if (messages.length > 0) {
        conversations.push({
          title: sessionData.chatTitle || "Untitled conversation",
          summary: sessionData.chatSummary || "",
          messageCount: sessionData.messageCount || messages.length,
          createdAt: sessionData.createdAt?.toDate?.()?.toISOString() || null,
          messages,
        });
      }
    }
  }

  return conversations;
}

/**
 * Send a batch of conversations to Claude for analysis.
 */
async function analyzeWithClaude(client, conversations) {
  const transcripts = conversations.map((conv, idx) => {
    const msgs = conv.messages
      .map((m) => `${m.role === "user" ? "USER" : "ASSISTANT"}: ${m.content}`)
      .join("\n");
    return `--- CONVERSATION ${idx + 1} (${conv.messages.length} messages) ---\n${msgs}`;
  });

  const prompt = `You are analyzing therapy/wellness chat conversations from a mobile app called GirlTalk.
Analyze the following ${conversations.length} conversations and provide structured insights.

${transcripts.join("\n\n")}

Respond with ONLY valid JSON in this exact format:
{
  "topics": [
    {"name": "Topic Name", "count": <number>, "exampleSnippets": ["short quote 1", "short quote 2"]}
  ],
  "painPoints": [
    {"description": "Description of the pain point", "frequency": <count>, "exampleSnippets": ["short quote"]}
  ],
  "sentiment": {
    "positive": <percentage 0-100>,
    "neutral": <percentage 0-100>,
    "negative": <percentage 0-100>
  },
  "engagement": {
    "avgMessageCount": <number>,
    "deepConversationRate": <percentage of conversations with 6+ user messages>
  },
  "keyInsights": ["insight 1", "insight 2", "insight 3"]
}

Guidelines:
- Topics: specific therapy/wellness themes (e.g. "Anxiety & Stress", "Relationship Issues", "Self-Esteem", "Grief & Loss", "Work Stress", "Family Conflict", "Depression", "Loneliness")
- Pain points: what users are struggling with or where the app experience falls short
- Example snippets: SHORT (under 15 words), anonymized — no names or identifying info
- Key insights: actionable observations for the product team`;

  const model = client.getGenerativeModel({ model: GEMINI_MODEL });
  const result = await model.generateContent(prompt);
  const text = result.response.text();
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) throw new Error("Gemini did not return valid JSON");
  return JSON.parse(jsonMatch[0]);
}


/**
 * Merge multiple batch results into one analysis document.
 */
function mergeBatchResults(results, sinceTimestamp, totalConversations) {
  const now = new Date();
  const dateKey = now.toISOString().split("T")[0];

  // Merge topics
  const topicMap = {};
  for (const result of results) {
    for (const topic of result.topics || []) {
      if (!topicMap[topic.name]) {
        topicMap[topic.name] = { name: topic.name, count: 0, exampleSnippets: [] };
      }
      topicMap[topic.name].count += topic.count;
      topicMap[topic.name].exampleSnippets.push(...(topic.exampleSnippets || []));
    }
  }
  const topics = Object.values(topicMap)
    .sort((a, b) => b.count - a.count)
    .map((t) => ({
      ...t,
      percentage: Math.round((t.count / totalConversations) * 1000) / 10,
      exampleSnippets: t.exampleSnippets.slice(0, 3),
    }));

  // Merge pain points
  const painMap = {};
  for (const result of results) {
    for (const pp of result.painPoints || []) {
      const key = pp.description.toLowerCase();
      if (!painMap[key]) {
        painMap[key] = { description: pp.description, frequency: 0, exampleSnippets: [] };
      }
      painMap[key].frequency += pp.frequency;
      painMap[key].exampleSnippets.push(...(pp.exampleSnippets || []));
    }
  }
  const painPoints = Object.values(painMap)
    .sort((a, b) => b.frequency - a.frequency)
    .map((p) => ({ ...p, exampleSnippets: p.exampleSnippets.slice(0, 3) }));

  // Average sentiment
  const sentimentSum = { positive: 0, neutral: 0, negative: 0 };
  for (const result of results) {
    if (result.sentiment) {
      sentimentSum.positive += result.sentiment.positive || 0;
      sentimentSum.neutral += result.sentiment.neutral || 0;
      sentimentSum.negative += result.sentiment.negative || 0;
    }
  }
  const n = results.length || 1;
  const sentiment = {
    positive: Math.round((sentimentSum.positive / n) * 10) / 10,
    neutral: Math.round((sentimentSum.neutral / n) * 10) / 10,
    negative: Math.round((sentimentSum.negative / n) * 10) / 10,
  };

  // Average engagement
  let totalMsgCount = 0;
  let deepCount = 0;
  for (const result of results) {
    if (result.engagement) {
      totalMsgCount += result.engagement.avgMessageCount || 0;
      deepCount += result.engagement.deepConversationRate || 0;
    }
  }
  const engagement = {
    avgMessageCount: Math.round((totalMsgCount / n) * 10) / 10,
    deepConversationRate: Math.round((deepCount / n) * 10) / 10,
  };

  // Deduplicate insights
  const allInsights = results.flatMap((r) => r.keyInsights || []);
  const keyInsights = [...new Set(allInsights)].slice(0, 10);

  return {
    dateKey,
    runDate: now,
    totalChatsAnalyzed: totalConversations,
    totalMessagesAnalyzed: totalConversations * Math.round(engagement.avgMessageCount),
    periodStart: sinceTimestamp,
    periodEnd: now,
    topics,
    painPoints,
    sentiment,
    engagement,
    keyInsights,
  };
}

/**
 * Save insights to Firestore and mark sessions as analyzed.
 */
async function saveInsights(result, newSessionIds) {
  const dateKey = result.dateKey || new Date().toISOString().split("T")[0];

  await db.doc(`chatInsights/${dateKey}`).set(result, { merge: true });

  const metaUpdate = { lastRunDate: dateKey };

  if (newSessionIds && newSessionIds.length > 0) {
    // arrayUnion adds the IDs to the list without duplicates
    metaUpdate.analyzedSessionIds = admin.firestore.FieldValue.arrayUnion(...newSessionIds);
  }

  await db.doc("chatInsights/_metadata").set(metaUpdate, { merge: true });
}
