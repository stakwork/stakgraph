import { Request, Response } from 'express';
import { db } from '../graph/neo4j.js';
import { generateText } from 'ai';
import { resolveLLMConfig } from '../aieo/src/index.js';
import * as asyncReqs from '../graph/reqs.js';

function buildPrompt(workflow_json: string): string {
  return `You are documenting a workflow. Given the following workflow JSON, produce concise documentation covering:
- Inputs: what data/parameters this workflow expects
- Logic: what the workflow does step by step
- Child workflows: any sub-workflows or external calls it makes
- Output: what it produces or returns

Workflow JSON:
${workflow_json}`;
}

export async function document_workflow(req: Request, res: Response) {
  const { node_key, ref_id } = req.body;
  if (!node_key && !ref_id) {
    res.status(400).json({ error: 'Missing node_key or ref_id' });
    return;
  }
  const workflow = await db.get_workflow_by_key(node_key, ref_id);
  if (!workflow) {
    res.status(400).json({ error: 'Workflow not found' });
    return;
  }
  const request_id = asyncReqs.startReq();
  res.json({ request_id, status: 'pending' });

  const llm = resolveLLMConfig({ model: req.body.model, apiKey: req.body.apiKey });
  (async () => {
    try {
      const result = await generateText({ model: llm.model, prompt: buildPrompt(workflow.workflow_json || JSON.stringify(workflow)) });
      const name = `Documentation for ${workflow.workflow_name || workflow.node_key}`;
      await db.upsert_workflow_documentation(workflow.ref_id, name, result.text);
      asyncReqs.finishReq(request_id, {
        documentation: result.text,
        usage: {
          inputTokens: result.usage?.inputTokens || 0,
          outputTokens: result.usage?.outputTokens || 0,
          totalTokens: result.usage?.totalTokens || 0,
        },
      });
    } catch (e: any) {
      asyncReqs.failReq(request_id, e.message || String(e));
    }
  })();
}

export async function document_workflows(req: Request, res: Response) {
  const request_id = asyncReqs.startReq();
  res.json({ request_id, status: 'pending' });

  const llm = resolveLLMConfig({ model: req.body.model, apiKey: req.body.apiKey });
  const model = llm.model;
  (async () => {
    try {
      const workflows = await db.get_all_workflows();
      let processed = 0, skipped = 0, totalInputTokens = 0, totalOutputTokens = 0;
      for (const workflow of workflows) {
        const existing = await db.get_workflow_documentation(workflow.node_key);
        if (existing) { skipped++; continue; }
        const result = await generateText({ model, prompt: buildPrompt(workflow.workflow_json || JSON.stringify(workflow)) });
        const name = `Documentation for ${workflow.workflow_name || workflow.node_key}`;
        await db.upsert_workflow_documentation(workflow.ref_id, name, result.text);
        totalInputTokens += result.usage?.inputTokens || 0;
        totalOutputTokens += result.usage?.outputTokens || 0;
        processed++;
        asyncReqs.updateReq(request_id, { processed, skipped, total: workflows.length });
      }
      asyncReqs.finishReq(request_id, {
        processed,
        skipped,
        usage: {
          inputTokens: totalInputTokens,
          outputTokens: totalOutputTokens,
          totalTokens: totalInputTokens + totalOutputTokens,
        },
      });
    } catch (e: any) {
      asyncReqs.failReq(request_id, e.message || String(e));
    }
  })();
}
