/*
  EimemesChat AI — Vercel Serverless Version
  Single provider: HuggingFace Llama
*/

'use strict';
require('dotenv').config();

const express = require('express');
const path    = require('path');
const app     = express();

app.use(express.json({ limit: '1mb' }));

// ── Logger ──────────────────────────────────────
const log = (tag, msg) => console.log(`[${new Date().toISOString()}] [${tag}] ${msg}`);

// ── Timeout helper ─────────────────────────────
function withTimeout(promise, ms, label) {
  let id;
  const timer = new Promise((_, reject) => {
    id = setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms);
  });
  return Promise.race([promise, timer]).finally(() => clearTimeout(id));
}

const PROVIDER_TIMEOUT = 25_000;

// ── HuggingFace Llama ──────────────────────────
async function tryHuggingFace(messages, abortSignal) {
  if (!process.env.HUGGINGFACE_API_KEY) {
    throw new Error('HUGGINGFACE_API_KEY not set in Vercel environment');
  }

  const fetchCall = fetch(
    'https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions',
    {
      method: 'POST',
      signal: abortSignal,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        max_tokens: 1024,
        messages: [
          {
            role: 'system',
            content: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant. Keep responses clear and well-formatted.',
          },
          ...messages.map(m => ({
            role: m.role === 'assistant' ? 'assistant' : 'user',
            content: m.content,
          })),
        ],
      }),
    }
  );

  const res = await withTimeout(fetchCall, PROVIDER_TIMEOUT, 'HuggingFace');
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`HuggingFace HTTP ${res.status}: ${body.slice(0, 160)}`);
  }
  const data = await res.json();
  if (!data?.choices?.[0]?.message?.content) {
    throw new Error('HuggingFace: unexpected response shape');
  }
  return { 
    reply: data.choices[0].message.content, 
    model: 'Llama (HF)' 
  };
}

// ── Generate title ─────────────────────────────
async function generateTitle(firstMessage, abortSignal) {
  if (!process.env.HUGGINGFACE_API_KEY) return null;

  const prompt = `Generate a very short title (max 6 words) for a conversation that starts with: "${firstMessage}". Output only the title, no quotes, no extra text.`;

  const fetchCall = fetch(
    'https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions',
    {
      method: 'POST',
      signal: abortSignal,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'meta-llama/Llama-3.1-8B-Instruct',
        max_tokens: 30,
        messages: [{ role: 'user', content: prompt }],
      }),
    }
  );

  try {
    const res = await withTimeout(fetchCall, 10000, 'TitleGen');
    if (!res.ok) return null;
    const data = await res.json();
    return data?.choices?.[0]?.message?.content?.trim() || null;
  } catch {
    return null;
  }
}

// ── API Route ──────────────────────────────────
app.post('/api/chat', async (req, res) => {
  const { message, history = [] } = req.body;

  if (!message?.trim()) {
    return res.status(400).json({ error: 'Message is required.' });
  }

  const messages = [
    ...history.filter(m => m.role && m.content).map(m => ({ 
      role: m.role, 
      content: m.content 
    })),
    { role: 'user', content: message.trim() },
  ];

  const ctrl = new AbortController();
  req.on('close', () => {
    if (!res.headersSent) ctrl.abort();
  });

  try {
    const result = await tryHuggingFace(messages, ctrl.signal);
    const response = { 
      reply: result.reply, 
      model: result.model 
    };

    // Generate title for first message
    if (history.length === 0) {
      const title = await generateTitle(message.trim(), ctrl.signal);
      if (title) response.title = title;
    }

    return res.json(response);
  } catch (err) {
    if (err.name === 'AbortError' || ctrl.signal.aborted) {
      return;
    }
    return res.status(503).json({
      error: 'AI service is temporarily unavailable. Please try again.',
      details: err.message,
    });
  }
});

// ── Health check ───────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    provider: process.env.HUGGINGFACE_API_KEY ? 'HuggingFace Llama' : 'No API key',
    timestamp: new Date().toISOString(),
  });
});

// ── Export for Vercel ──────────────────────────
module.exports = app;
