import { Command } from "commander";
import { runNewAgent } from "./commands/new_agent.js";
import { runWhoami } from "./commands/whoami.js";
import { runCommit } from "./commands/commit.js";
import { runTag } from "./commands/tag.js";
import { runPush } from "./commands/push.js";
import { spawnGit } from "./git.js";

/**
 * Top-level dispatch.
 *
 * Sphinx-only subcommands (`new-agent`, `whoami`) are routed through commander
 * for help text and arg validation. Everything that wraps real git
 * (`commit`, `tag`, `push`, and the catch-all passthrough) bypasses commander
 * so git's own flags are never mis-parsed.
 */
const SPHINX_SUBS = new Set(["new-agent", "whoami", "help", "--help", "-h"]);

export async function run(argv: string[]): Promise<void> {
  const sub = argv[0];

  if (sub === undefined) {
    // bare `sphinx-git` — show our own help, not git's
    printHelp();
    process.exit(1);
  }

  if (SPHINX_SUBS.has(sub)) {
    runSphinxSubcommand(argv);
    return;
  }

  if (sub === "commit") {
    await runCommit(argv.slice(1));
    return;
  }
  if (sub === "tag") {
    await runTag(argv.slice(1));
    return;
  }
  if (sub === "push") {
    await runPush(argv.slice(1));
    return;
  }

  // Everything else: pure passthrough to git, no env/-c modification.
  await spawnGit(argv);
}

function runSphinxSubcommand(argv: string[]): void {
  const program = new Command();

  program
    .name("sphinx-git")
    .description(
      "Drop-in git replacement that signs commits/tags with a per-agent ed25519 key.",
    )
    .version("0.1.0");

  program
    .command("new-agent")
    .description("Configure this sandbox with a new agent identity (overwrites existing).")
    .argument("<hex_privkey>", "64-char hex ed25519 private key (32 bytes)")
    .argument("<child>", "non-negative integer child index assigned by Hive")
    .action((hexPriv: string, child: string) => {
      runNewAgent([hexPriv, child]);
    });

  program
    .command("whoami")
    .description("Print the configured agent's child index, pubkey, and git author string.")
    .action(() => {
      runWhoami();
    });

  // commander doesn't know about commit/tag/push (they're handled before us),
  // but we do want it to know they exist so `--help` lists them.
  program.command("commit [args...]").description("Run `git commit` with agent identity + SSH signature.");
  program.command("tag [args...]").description("Run `git tag`; SSH-signs annotated tags.");
  program.command("push [args...]").description("Refuse to push unsigned HEAD; otherwise run `git push`.");

  program.parse(argv, { from: "user" });
}

function printHelp(): void {
  process.stderr.write(
    "sphinx-git: drop-in git wrapper that signs commits with a per-agent key.\n" +
      "\n" +
      "Sphinx subcommands:\n" +
      "  sphinx-git new-agent <hex_privkey> <child>   configure agent identity\n" +
      "  sphinx-git whoami                            print configured identity\n" +
      "\n" +
      "Wrapped git subcommands (signed):\n" +
      "  sphinx-git commit ...args                    git commit, SSH-signed\n" +
      "  sphinx-git tag ...args                       git tag (annotated tags signed)\n" +
      "  sphinx-git push ...args                      refuses unsigned HEAD\n" +
      "\n" +
      "All other subcommands are passed through to git unchanged.\n",
  );
}
