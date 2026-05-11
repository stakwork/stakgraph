// List virtual keys in Bifrost. Pass --delete to wipe them all.
//
//   tsx list-vks.ts
//   tsx list-vks.ts --delete

import { deleteVirtualKey, listVirtualKeys, ping } from "./lib.js";

async function main() {
  await ping();
  const vks = await listVirtualKeys();
  if (process.argv.includes("--delete")) {
    for (const vk of vks) {
      console.error(`deleting ${vk.id} (${vk.name})`);
      await deleteVirtualKey(vk.id);
    }
    console.error(`deleted ${vks.length} virtual keys`);
    return;
  }
  console.log(JSON.stringify(vks.map((v) => ({
    id: v.id,
    name: v.name,
    value: v.value,
    is_active: v.is_active,
    providers: v.provider_configs?.map((p: any) => p.provider) ?? [],
  })), null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
