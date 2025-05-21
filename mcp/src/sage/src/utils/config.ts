import * as fs from "fs";
import * as path from "path";

export interface Config {
  github: {
    owner: string;
    repo: string;
    token: string;
  };
  codeSpaceURL: string;
  "2b_base_url": string;
  secret: string;
  stakwork_api_key: string;
  workflow_id: string;
  data_dir?: string;
  dry_run?: boolean;
}

export function loadConfig(configPath: string = "sage_config.json"): Config {
  try {
    const configFilePath = path.resolve(process.cwd(), configPath);
    const configData = fs.readFileSync(configFilePath, "utf8");
    return JSON.parse(configData) as Config;
  } catch (error) {
    // console.error(`Error loading config from ${configPath}:`, error);
    throw new Error(`Failed to load configuration: ${error}`);
  }
}
