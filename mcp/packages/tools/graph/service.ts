import * as path from "path";
import * as yaml from "js-yaml";
import { ContainerConfig, Neo4jNode, Service, ServiceParser } from "./types.js";
import { Language } from "./types.js";

class TsParser implements ServiceParser {
  pkgFileName = "package.json";
  envRegex = /process\.env\.(\w+)/g;

  build(pkgFile: Neo4jNode): Service {
    const body = pkgFile.properties.body;
    const dir = path.dirname(pkgFile.properties.file);

    if (
      !body ||
      body.trim() === "" ||
      body === "undefined" ||
      body === "null"
    ) {
      console.warn(
        `Invalid package.json body for ${pkgFile.properties.file}: ${body}`
      );
      return this.createDefaultService(dir, pkgFile.properties.file);
    }

    try {
      const pkg = JSON.parse(body);
      return {
        name: pkg.name || path.basename(dir),
        language: "javascript",
        dev: true,
        scripts: {
          start: pkg.scripts?.start || "npm start",
          install: "npm install",
          build: pkg.scripts?.build || "npm run build",
          test: pkg.scripts?.test || "npm test",
        },
        env: {},
        pkgFile: pkgFile.properties.file,
      };
    } catch (error) {
      console.error(
        `Failed to parse package.json for ${pkgFile.properties.file}:`,
        error
      );
      return this.createDefaultService(dir, pkgFile.properties.file);
    }
  }

  private createDefaultService(dir: string, filePath: string): Service {
    return {
      name: path.basename(dir),
      language: "javascript",
      dev: true,
      scripts: {
        install: "npm install",
        start: "npm start",
        build: "npm run build",
        test: "npm test",
      },
      env: {},
      pkgFile: filePath,
    };
  }
}

class GoParser implements ServiceParser {
  pkgFileName = "go.mod";
  envRegex = /os\.(?:Getenv|LookupEnv)\("([^"]+)"\)/g;

  build(pkgFile: Neo4jNode): Service {
    const dir = path.dirname(pkgFile.properties.file);
    const serviceName = path.basename(dir);

    return {
      name: serviceName,
      language: "go",
      dev: true,
      scripts: {
        install: "go mod tidy",
        build: `go build -o ${serviceName}`,
        start: `./${serviceName}`,
        test: "go test ./...",
      },
      env: {},
      pkgFile: pkgFile.properties.file,
    };
  }
}
class RustParser implements ServiceParser {
  pkgFileName = "Cargo.toml";
  envRegex = /std::env::var\("([^"]+)"\)/g;

  build(pkgFile: Neo4jNode): Service {
    const dir = path.dirname(pkgFile.properties.file);
    const serviceName = path.basename(dir);

    return {
      name: serviceName,
      language: "rust",
      dev: true,
      scripts: {
        install: "cargo build",
        start: `cargo run`,
        build: "cargo build --release",
        test: "cargo test",
      },
      env: {},
      pkgFile: pkgFile.properties.file,
    };
  }
}

class RubyParser implements ServiceParser {
  pkgFileName = "Gemfile";
  envRegex = /ENV\[['"]([^'"]+)['"]\]/g;

  build(pkgFile: Neo4jNode): Service {
    const dir = path.dirname(pkgFile.properties.file);
    const serviceName = path.basename(dir);

    return {
      name: serviceName,
      language: "ruby",
      dev: true,
      scripts: {
        install: "bundle install",
        start: `bundle exec rails server`,
        build: "echo 'No build step for Ruby/Rails by default'",
        test: "bundle exec rake test",
      },
      env: {},
      pkgFile: pkgFile.properties.file,
    };
  }
}

const serviceParsers: ServiceParser[] = [
  new TsParser(),
  new GoParser(),
  new RustParser(),
  new RubyParser(),
];

export function parseServiceFile(
  pkgFile: string,
  body: string,
  language: Language
): Service {
  const parser = serviceParsers.find((p) => pkgFile.endsWith(p.pkgFileName));

  if (parser) {
    const neo4jNode: Neo4jNode = {
      properties: {
        body,
        file: pkgFile,
        name: path.basename(pkgFile),
        start: 0,
        end: body.length,
      },
      labels: ["File"],
    };
    return parser.build(neo4jNode);
  }

  //Fallback: return bare minimum
  return {
    name: path.basename(path.dirname(pkgFile)),
    language,
    pkgFile,
    scripts: {},
    env: {},
    dev: false,
  };
}

function extractEnvVarNames(body: string, regexes: RegExp[]): string[] {
  if (!body) return [];
  let allMatches: string[] = [];
  for (const regex of regexes) {
    const matches = [...body.matchAll(regex)].map((match) => match[1]);
    allMatches = allMatches.concat(matches);
  }
  return allMatches;
}

export function generate_services_config(
  pkgFiles: Neo4jNode[],
  allFiles: Neo4jNode[],
  envVarNodes: Neo4jNode[]
): { services: any[]; containers: ContainerConfig[] } {
  const serviceMap = new Map<string, any>();

  // parse package files
  for (const file of pkgFiles) {
    console.log("===> file", file.properties.name);
    const parser = serviceParsers.find((p) =>
      file.properties.name.endsWith(p.pkgFileName)
    );
    console.log("===> parser", parser);
    if (parser) {
      const service = parser.build(file);
      if (service.name) {
        console.log("===> service", service.name);
        serviceMap.set(service.name, service);
      } else {
        console.log("===> no service name", file.properties.name);
      }
    }
  }

  console.log("===> serviceMap", serviceMap);

  const dcFiles = allFiles.filter(
    (f) =>
      f.properties.name.endsWith("docker-compose.yml") ||
      f.properties.name.endsWith("docker-compose.yaml")
  );
  let containers: ContainerConfig[] = [];
  for (const dcFile of dcFiles) {
    const body = dcFile.properties.body;
    if (
      !body ||
      body.trim() === "" ||
      body === "undefined" ||
      body === "null"
    ) {
      console.warn(
        `Invalid docker-compose body for ${dcFile.properties.file}: ${body}`
      );
      continue;
    }

    const found = extractContainersFromComposeBody(body);
    containers = containers.concat(found);
  }

  const envRegexes = serviceParsers.map((p) => p.envRegex);
  for (const envVarNode of envVarNodes) {
    const nodeFile = envVarNode.properties.file;
    if (!nodeFile) continue;

    const body = envVarNode.properties.body;
    if (!body || body === "undefined" || body === "null") continue;

    const varNames = extractEnvVarNames(body, envRegexes);
    if (varNames.length === 0) continue;

    let bestMatchService: any = null;
    let longestMatchPath = "";

    for (const service of serviceMap.values()) {
      if (service.language !== "unknown" && service.pkgFile) {
        const serviceDir = path.dirname(service.pkgFile);
        if (
          nodeFile.startsWith(serviceDir) &&
          serviceDir.length > longestMatchPath.length
        ) {
          longestMatchPath = serviceDir;
          bestMatchService = service;
        }
      }
    }

    if (bestMatchService) {
      for (const varName of varNames) {
        if (!bestMatchService.env[varName]) {
          bestMatchService.env[varName] = "";
        }
      }
    }
  }

  const finalServices: any[] = [];
  for (const service of serviceMap.values()) {
    const { pkgFile, ...finalService } = service;
    finalServices.push(finalService);
  }

  return { services: finalServices, containers };
}

export function extractContainersFromComposeBody(
  composeBody: string
): ContainerConfig[] {
  let containers: ContainerConfig[] = [];
  try {
    const doc = yaml.load(composeBody) as any;
    if (doc && doc.services) {
      for (const [name, svc] of Object.entries<any>(doc.services)) {
        if (!svc.build) {
          const config = yaml.dump(svc, { noRefs: true });
          containers.push({ name, config });
        }
      }
    }
  } catch (e) {
    console.error(`Failed to parse docker-compose content:`, e);
  }
  return containers;
}
export async function extractContainersFromCompose(
  composeFilePath: string
): Promise<ContainerConfig[]> {
  const fs = await import("fs/promises");
  try {
    const body = await fs.readFile(composeFilePath, "utf8");
    return extractContainersFromComposeBody(body);
  } catch (e) {
    console.error(`Failed to read docker-compose file: ${composeFilePath}`, e);
    return [];
  }
}
