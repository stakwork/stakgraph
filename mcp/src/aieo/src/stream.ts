import { ModelMessage, streamText, generateObject, ToolSet } from "ai";
import {
  Provider,
  getModel,
  getProviderOptions,
  ThinkingSpeed,
} from "./provider.js";
import { z } from "zod";

interface CallModelOptions {
  provider: Provider;
  apiKey: string;
  messages: ModelMessage[];
  tools?: ToolSet;
  parser?: (fullResponse: string) => void;
  thinkingSpeed?: ThinkingSpeed;
  cwd?: string;
  executablePath?: string;
}

export async function callModel(opts: CallModelOptions): Promise<string> {
  const {
    provider,
    apiKey,
    messages,
    tools,
    parser,
    thinkingSpeed,
    cwd,
    executablePath,
  } = opts;
  const model = await getModel(provider, apiKey, cwd, executablePath);
  const providerOptions = getProviderOptions(provider, thinkingSpeed);
  console.log(`Calling ${provider} with options:`, providerOptions);
  const result = streamText({
    model,
    tools,
    messages,
    temperature: 0,
    providerOptions: providerOptions as any,
  });
  let fullResponse = "";
  for await (const part of result.fullStream) {
    console.log(part);
    switch (part.type) {
      case "error":
        throw part.error;
      case "text-delta":
        if (parser) {
          parser(fullResponse);
        }
        fullResponse += part.text;
        break;
    }
  }
  return fullResponse;
}

const ppp =
  "## How Authentication Works in the Hive Repository\n\nThe authentication system in this repository is built around **NextAuth.js** with multiple authentication modes and role-based access control. Here's how it works:\n\n### Core Authentication Architecture\n\n**Main Configuration**: `src/lib/auth/nextauth.ts`\n- Uses NextAuth.js with custom providers and callbacks\n- Supports both database and JWT-based sessions\n- Integrates with Prisma for user/account management\n\n### Authentication Providers\n\nThe system supports **dual authentication modes**:\n\n1. **GitHub OAuth Provider** (Production):\n   - Uses GitHub OAuth for real user authentication\n   - Stores encrypted access tokens using `EncryptionService`\n   - Handles GitHub profile data and permissions\n   - Configured via `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`\n\n2. **Mock/Credentials Provider** (Development):\n   - Activated when `POD_URL` environment variable is set\n   - Allows development authentication with any username\n   - Creates mock users with `@mock.dev` email addresses\n   - Uses JWT sessions instead of database sessions\n\n### Role-Based Access Control (RBAC)\n\n**Roles System**: `src/lib/auth/roles.ts`\n- **Role Hierarchy** (1-6 levels):\n  - VIEWER (1) - Basic read access\n  - STAKEHOLDER (2) - Limited participation\n  - DEVELOPER (3) - Code and development access\n  - PM (4) - Project management capabilities\n  - ADMIN (5) - Administrative functions\n  - OWNER (6) - Full workspace control\n\n**Permission Functions**:\n- `hasRoleLevel()` - Checks if user meets minimum role requirement\n- `isWorkspaceRole()` - Validates role values\n- `isAssignableMemberRole()` - Checks assignable roles\n\n### Workspace-Based Authorization\n\n**Workspace Resolution**: `src/lib/auth/workspace-resolver.ts`\n- Users belong to workspaces with specific roles\n- `validateUserWorkspaceAccess()` - Validates user access to specific workspaces\n- `resolveUserWorkspaceRedirect()` - Handles workspace routing based on user permissions\n- Default workspace assignment for new users\n\n**Client-Side Access Control**: `src/hooks/useWorkspaceAccess.ts`\n- Provides React hooks for permission checking\n- Methods like `canManage()`, `hasMinimumRole()`, `requiresRole()`\n- Permission levels: none, read, write, admin, owner\n\n### Authentication Flow\n\n1. **Sign In Process**: \n   - GitHub OAuth redirects to `/api/auth/signin`\n   - NextAuth processes authentication via `/api/auth/[...nextauth]/route.ts`\n   - User profile created/updated in database\n   - GitHub access tokens encrypted and stored\n   - Default workspace assigned if needed\n\n2. **Session Management**:\n   - Session callbacks in `nextauth.ts` attach user data and workspace info\n   - JWT tokens used for mock authentication\n   - Database sessions used for GitHub OAuth\n\n3. **Access Control**:\n   - Every workspace route checks user permissions\n   - Role hierarchy determines available actions\n   - UI components conditionally render based on permissions\n\n### Key Components\n\n**Frontend**:\n- `src/components/LoginButton.tsx` - Authentication UI component\n- `src/contexts/WorkspaceContext.tsx` - Provides workspace and role context\n- `src/hooks/useWorkspaceAccess.ts` - Permission checking hooks\n\n**Backend**:\n- `src/app/api/auth/[...nextauth]/route.ts` - NextAuth API endpoint\n- `src/app/api/auth/revoke-github/route.ts` - GitHub token revocation\n- Various API routes check session and workspace permissions\n\n### Security Features\n\n1. **Token Encryption**: GitHub access tokens encrypted using `src/lib/encryption/`\n2. **Workspace Isolation**: Users can only access workspaces they're members of\n3. **Role Validation**: All operations check minimum required roles\n4. **Session Management**: Secure session handling with NextAuth\n5. **Development Safety**: Mock authentication for development environments\n\n### Testing\n\nAuthentication is tested in:\n- `src/__tests__/e2e/auth.spec.ts` - End-to-end authentication tests\n- `src/__tests__/integration/api/swarm-authorization.test.ts` - API authorization tests\n- `src/__tests__/unit/lib/auth/roles.test.ts` - Role system unit tests\n\nThis authentication system provides a robust, multi-modal approach supporting both development workflows and production security requirements while maintaining role-based access control across workspaces.";

interface GenerateObjectArgs {
  provider: Provider;
  apiKey: string;
  prompt: string;
  schema: z.ZodObject<any>;
}

export async function callGenerateObject(args: GenerateObjectArgs) {
  const model = await getModel(args.provider, args.apiKey);
  console.log("CALLING CLAUDE WITH PROMPT");
  const { object } = await generateObject({
    model,
    schema: args.schema,
    prompt: args.prompt,
  });
  console.log(object);
  return object;
}
