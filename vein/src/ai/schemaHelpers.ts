import { z } from "zod";

// ── Schema helpers (reused from server.ts pattern) ─────────────────────────

export interface FieldDesc {
  name: string;
  kind: "string" | "number" | "boolean" | "enum" | "json";
  required: boolean;
  default?: unknown;
  enumValues?: string[];
}

export function zodToFields(schema: z.ZodTypeAny): FieldDesc[] {
  const shape = getObjectShape(schema);
  if (!shape) return [];
  return Object.entries(shape).map(([name, s]) =>
    describeField(name, s as z.ZodTypeAny),
  );
}

function getObjectShape(s: z.ZodTypeAny): Record<string, z.ZodTypeAny> | null {
  const def = s._def;
  if (def.typeName === "ZodObject") return (def as any).shape();
  if (def.typeName === "ZodEffects") return getObjectShape(def.schema);
  return null;
}

function describeField(name: string, s: z.ZodTypeAny): FieldDesc {
  let required = true;
  let defaultVal: unknown;
  let inner = s;
  for (;;) {
    const def = inner._def;
    if (def.typeName === "ZodOptional") {
      required = false;
      inner = def.innerType;
    } else if (def.typeName === "ZodDefault") {
      required = false;
      defaultVal = def.defaultValue();
      inner = def.innerType;
    } else if (def.typeName === "ZodNullable") {
      required = false;
      inner = def.innerType;
    } else break;
  }
  const typeName = inner._def.typeName as string;
  if (typeName === "ZodEnum")
    return { name, kind: "enum", required, default: defaultVal, enumValues: inner._def.values };
  if (typeName === "ZodString") return { name, kind: "string", required, default: defaultVal };
  if (typeName === "ZodNumber") return { name, kind: "number", required, default: defaultVal };
  if (typeName === "ZodBoolean") return { name, kind: "boolean", required, default: defaultVal };
  return { name, kind: "json", required, default: defaultVal };
}
