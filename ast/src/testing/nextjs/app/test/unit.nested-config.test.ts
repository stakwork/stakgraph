import { describe, it, expect } from 'vitest';
import { authOptions, apiConfig, routerConfig } from '../../lib/nested-config';

describe('Nested Config Tests', () => {
  it('should call signIn callback from authOptions', async () => {
    const signInCallback = authOptions.callbacks.signIn;
    
    const result = await signInCallback({
      user: { email: 'test@example.com', id: '1', role: 'user' },
      account: null,
      profile: null,
    });
    
    expect(result).toBe(true);
  });
  
  it('should call jwt callback from authOptions', async () => {
    const jwtCallback = authOptions.callbacks.jwt;
    
    const token = await jwtCallback({
      token: {},
      user: { id: '123', role: 'admin' },
      account: null,
    });
    
    expect(token.userId).toBe('123');
  });
  
  it('should call session callback', async () => {
    const sessionCallback = authOptions.callbacks.session;
    
    const session = await sessionCallback({
      session: { user: {} },
      token: { userId: '456', role: 'user' },
    });
    
    expect(session.user.id).toBe('456');
  });
  
  it('should call nested api endpoint: users.list', async () => {
    const listUsers = apiConfig.endpoints.users.list;
    const users = await listUsers();
    expect(users).toBeDefined();
  });
  
  it('should call nested api endpoint: users.create', async () => {
    const createUser = apiConfig.endpoints.users.create;
    const user = await createUser({ name: 'John' });
    expect(user).toBeDefined();
  });
  
  it('should call deeply nested handler', async () => {
    const profileHandler = routerConfig.routes.settings.nested.profile.handler;
    const result = await profileHandler();
    expect(result.title).toBe('Profile Settings');
  });
  
  it('should call middleware auth function', async () => {
    const authMiddleware = apiConfig.middleware.auth;
    
    await expect(
      authMiddleware({ headers: {} })
    ).rejects.toThrow('Unauthorized');
  });
  
  it('should access dashboard handler', async () => {
    const dashboardHandler = routerConfig.routes.dashboard.handler;
    const result = await dashboardHandler();
    expect(result.title).toBe('Dashboard');
  });
});
