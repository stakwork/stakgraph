import { test, expect } from '../testkit.js';
import path from 'path';
import fs from 'fs';
import os from 'os';
import {
  scanSkills,
  clearSkillsCache,
  enabledEntries,
  renderSkillIndex,
  listPack,
  loadSkill,
  SKILLS,
} from "../repo/skills.js";
import { get_tools } from "../repo/tools.js";

/**
 * Build a throwaway skills root:
 *
 *   solo/SKILL.md              a bare skill
 *   packA/PACK.md              pack preamble + description
 *   packA/{alpha,shared}/SKILL.md
 *   packB/{beta,shared}/SKILL.md   ("shared" collides with packA's on purpose)
 *   empty/                     no SKILL.md anywhere -> not an entry
 */
async function createSkillsRoot(): Promise<string> {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'skills-test-'));
  const write = async (rel: string, body: string) => {
    const p = path.join(root, rel);
    await fs.promises.mkdir(path.dirname(p), { recursive: true });
    await fs.promises.writeFile(p, body);
  };
  const skill = (name: string, desc: string, body = "do the thing") =>
    `---\nname: ${name}\ndescription: ${desc}\n---\n\n${body}\n`;

  await write('solo/SKILL.md', skill('solo', 'A standalone skill'));
  await write('packA/PACK.md', `---\nname: packA\ndescription: Pack A does A things\ntags: [grouped]\n---\n\nSHARED PACK A CONTEXT\n`);
  await write('packA/alpha/SKILL.md', skill('alpha', 'Alpha skill', 'ALPHA BODY'));
  await write('packA/shared/SKILL.md', skill('shared', 'Shared name in A'));
  await write('packB/PACK.md', `---\nname: packB\ndescription: Pack B does B things\ntags: [grouped, other]\n---\n\nSHARED PACK B CONTEXT\n`);
  await write('packB/beta/SKILL.md', skill('beta', 'Beta skill'));
  await write('packB/shared/SKILL.md', skill('shared', 'Shared name in B'));
  await fs.promises.mkdir(path.join(root, 'empty'), { recursive: true });

  return root;
}

test.describe("skills loader", () => {
  let root: string;
  let prevRoot: string | undefined;
  let prevNoCache: string | undefined;

  test.beforeEach(async () => {
    root = await createSkillsRoot();
    prevRoot = process.env.SKILLS_ROOT;
    prevNoCache = process.env.SKILLS_NO_CACHE;
    process.env.SKILLS_ROOT = root;
    process.env.SKILLS_NO_CACHE = "1";
    clearSkillsCache();
  });

  test.afterEach(async () => {
    if (prevRoot === undefined) delete process.env.SKILLS_ROOT;
    else process.env.SKILLS_ROOT = prevRoot;
    if (prevNoCache === undefined) delete process.env.SKILLS_NO_CACHE;
    else process.env.SKILLS_NO_CACHE = prevNoCache;
    clearSkillsCache();
    await fs.promises.rm(root, { recursive: true, force: true });
  });

  test("classifies skills vs packs and skips dirs with no SKILL.md", async () => {
    const entries = scanSkills();
    expect(entries.map((e) => e.name)).toEqual(['packA', 'packB', 'solo']);
    expect(entries.find((e) => e.name === 'solo')?.kind).toBe('skill');

    const packA = entries.find((e) => e.name === 'packA');
    expect(packA?.kind).toBe('pack');
    expect(packA?.description).toBe('Pack A does A things');
    if (packA?.kind === 'pack') {
      expect(packA.skills.map((s) => s.qualified)).toEqual(['packA/alpha', 'packA/shared']);
    }
  });

  test("index lists only enabled entries and drops unknown names", async () => {
    const index = renderSkillIndex(
      enabledEntries({ packA: true, solo: true, packB: false, 'not-installed': true }),
    );
    expect(index).toContain('packA [pack, 2 skills]: Pack A does A things');
    expect(index).toContain('solo: A standalone skill');
    expect(index).not.toContain('packB');
    expect(index).not.toContain('not-installed');
    // A pack contributes one line, never its member skills.
    expect(index).not.toContain('alpha');
  });

  test("a pack-qualified flag enables just that skill", async () => {
    const index = renderSkillIndex(enabledEntries({ 'packA/alpha': true }));
    expect(index).toContain('packA/alpha: Alpha skill');
    expect(index).not.toContain('[pack,');
    expect(index).not.toContain('packA/shared');
  });

  test("no enabled skills produces no prompt block", async () => {
    expect(renderSkillIndex(enabledEntries({}))).toBe('');
    expect(renderSkillIndex(enabledEntries(undefined))).toBe('');
    expect(renderSkillIndex(enabledEntries({ packA: false }))).toBe('');
  });

  test("list_skills expands a pack and reports unknown packs", async () => {
    expect(listPack('packA')?.map((s) => s.qualified)).toEqual(['packA/alpha', 'packA/shared']);
    expect(listPack('nope')).toBeNull();
  });

  test("load_skill returns body plus the pack preamble", async () => {
    const r: any = loadSkill('packA/alpha');
    expect(r.error).toBeUndefined();
    expect(r.name).toBe('packA/alpha');
    expect(r.packName).toBe('packA');
    expect(r.body).toContain('ALPHA BODY');
    // Preamble is the PACK.md body with frontmatter stripped.
    expect(r.preamble).toBe('SHARED PACK A CONTEXT');
    expect(r.preamble).not.toContain('description:');
  });

  test("an unambiguous bare name resolves without its pack prefix", async () => {
    const r: any = loadSkill('alpha');
    expect(r.error).toBeUndefined();
    expect(r.name).toBe('packA/alpha');
  });

  test("a name colliding across packs errors with candidates instead of guessing", async () => {
    const r: any = loadSkill('shared');
    expect(r.error).toContain('ambiguous');
    expect(r.candidates).toEqual(['packA/shared', 'packB/shared']);
    expect(r.body).toBeUndefined();
  });

  test("unknown skill errors and points at list_skills", async () => {
    const r: any = loadSkill('nonexistent');
    expect(r.error).toContain('not found');
    expect(r.error).toContain('list_skills');
  });

  test("inline skills still resolve and take precedence over disk", async () => {
    const r: any = loadSkill('mermaid');
    expect(r.error).toBeUndefined();
    expect(r.body).toBe(SKILLS['mermaid']);
    expect(r.preamble).toBeUndefined();
  });

  test("a tag enables every pack carrying it", async () => {
    const entries = enabledEntries({ grouped: true });
    expect(entries.map((e) => e.name)).toEqual(['packA', 'packB']);
    // Still one index line per pack — a tag is a shorthand, not a 4th nesting level.
    const index = renderSkillIndex(entries);
    expect(index).toContain('packA [pack, 2 skills]');
    expect(index).toContain('packB [pack, 2 skills]');
    expect(index).not.toContain('grouped');
    // A one-skill pack reads "1 skill", not "1 skills".
    expect(renderSkillIndex([{ ...(entries[0] as any), skills: [(entries[0] as any).skills[0]] }]))
      .toContain('[pack, 1 skill]');
  });

  test("a tag matching one pack enables only that pack", async () => {
    expect(enabledEntries({ other: true }).map((e) => e.name)).toEqual(['packB']);
  });

  test("tag and explicit name together do not duplicate", async () => {
    expect(enabledEntries({ grouped: true, packA: true }).map((e) => e.name))
      .toEqual(['packA', 'packB']);
  });

  test("an unknown key matches neither name nor tag", async () => {
    expect(enabledEntries({ 'no-such-group': true })).toEqual([]);
  });

  test("bare skills carry no tags and are unaffected by tag selection", async () => {
    expect(enabledEntries({ grouped: true }).some((e) => e.name === 'solo')).toBe(false);
  });

  test("skills tools are registered only when the caller enabled something", async () => {
    const names = async (skills?: Record<string, boolean>) =>
      Object.keys(
        await get_tools('/tmp', '', undefined, {}, 'openai', undefined, undefined,
          undefined, undefined, undefined, undefined, undefined, undefined, skills),
      ).filter((n) => n === 'list_skills' || n === 'load_skill');

    // A run that never asked for skills pays no tool-schema tokens.
    expect(await names(undefined)).toEqual([]);
    expect(await names({})).toEqual([]);
    expect(await names({ packA: false })).toEqual([]);
    // Enabling anything opens the discovery path to everything installed.
    expect(await names({ packA: true })).toEqual(['list_skills', 'load_skill']);
  });

  test("a missing skills root is an empty index, not an error", async () => {
    process.env.SKILLS_ROOT = path.join(root, 'does-not-exist');
    clearSkillsCache();
    expect(scanSkills()).toEqual([]);
    expect(renderSkillIndex(enabledEntries({ packA: true }))).toBe('');
  });
});
