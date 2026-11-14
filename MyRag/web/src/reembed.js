import { api } from "./api.js";

export function embedIsStale(machine, models) {
  if (!machine || !models) return false;
  const configured = models.embeddings || [];
  return (
    machine.embed_dim !== models.embed_dim ||
    !configured.some((m) => m.id === machine.embed_model_id)
  );
}

export function staleError(err) {
  return err?.status === 409 && err?.payload?.detail?.code === "embed_model_stale";
}

/**
 * Ask the user to re-embed a machine whose previous embed model/dimension is gone,
 * then switch it to the server default. Returns the updated machine, or null when the
 * user cancels.
 */
export async function confirmReembed(machine, models) {
  const modelId = models?.default_embed_model;
  if (!modelId) {
    throw new Error("The server has no default embed model configured.");
  }

  const model = (models?.embeddings || []).find((m) => m.id === modelId);
  const question =
    `This machine used "${machine.embed_model_id}" (dim ${machine.embed_dim}), which is ` +
    `no longer configured. The server now expects dim ${models.embed_dim}.\n\n` +
    `Re-embed all ${machine.chunk_count} chunks with ${modelId} (dim ${model?.dim ?? "?"})?\n\n` +
    "This calls the embedding API again and may cost money.";

  if (!window.confirm(question)) return null;

  const updated = await api.updateMachine(machine.id, {
    embed_model_id: modelId,
    reembed: true,
  });
  return updated;
}