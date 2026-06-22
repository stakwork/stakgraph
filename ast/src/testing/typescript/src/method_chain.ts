// @ast node: Class "StoreRecord"
class StoreRecord {
  // @ast node: Function "process"
  process(): string {
    return "ok";
  }
}

// @ast node: Class "RecordStore"
class RecordStore {
  // @ast node: Function "loadRecord"
  async loadRecord(id: string): Promise<StoreRecord> {
    return new StoreRecord();
  }
}

// @ast node: Var "store"
const store = new RecordStore();

// @ast node: Function "handleLoad"
// @ast edge: Calls -> Function "loadRecord" "method_chain.ts"
// @ast edge: Calls -> Function "process" "method_chain.ts"
async function handleLoad(id: string): Promise<string> {
  const record = await store.loadRecord(id);
  return record.process();
}
