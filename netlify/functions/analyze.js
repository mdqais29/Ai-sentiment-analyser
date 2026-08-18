const vader = require("vader-sentiment");
const { analyzeText } = require("./lib/sentiment");

const SAMPLE_TEXTS = [
  "I absolutely love this product! Best purchase I've ever made.",
  "This is the worst experience I've had. Completely disappointed.",
  "The package arrived on Tuesday. It was a standard delivery.",
  "Not bad, but could definitely be improved in several areas.",
  "I'm thrilled with the customer support — they went above and beyond!",
];

const HEADERS = {
  "Content-Type": "application/json",
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

exports.handler = async (event) => {
  const method = event.httpMethod;

  if (method === "OPTIONS") {
    return { statusCode: 204, headers: HEADERS, body: "" };
  }

  if (method === "GET") {
    const path = event.path || "";
    if (path.endsWith("/health")) {
      return {
        statusCode: 200,
        headers: HEADERS,
        body: JSON.stringify({ status: "ok", service: "ai-sentiment-analyser" }),
      };
    }
    return {
      statusCode: 200,
      headers: HEADERS,
      body: JSON.stringify({ samples: SAMPLE_TEXTS }),
    };
  }

  if (method === "POST") {
    let payload = {};
    try {
      payload = JSON.parse(event.body || "{}");
    } catch {
      return {
        statusCode: 400,
        headers: HEADERS,
        body: JSON.stringify({ error: "Invalid JSON body" }),
      };
    }

    const text = (payload.text || "").trim();
    const texts = payload.texts || [];

    try {
      if (texts.length > 0) {
        const results = texts
          .filter((item) => String(item).trim())
          .map((item) => analyzeText(String(item), vader));
        return {
          statusCode: 200,
          headers: HEADERS,
          body: JSON.stringify({ results }),
        };
      }

      if (text) {
        return {
          statusCode: 200,
          headers: HEADERS,
          body: JSON.stringify(analyzeText(text, vader)),
        };
      }

      return {
        statusCode: 400,
        headers: HEADERS,
        body: JSON.stringify({ error: "Provide 'text' or 'texts' in request body" }),
      };
    } catch (error) {
      const statusCode = error.message === "Text cannot be empty" ? 400 : 500;
      return {
        statusCode,
        headers: HEADERS,
        body: JSON.stringify({ error: error.message || "Analysis failed" }),
      };
    }
  }

  return {
    statusCode: 404,
    headers: HEADERS,
    body: JSON.stringify({ error: "Not found" }),
  };
};
