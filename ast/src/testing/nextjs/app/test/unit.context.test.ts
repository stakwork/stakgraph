// @ts-nocheck
import { useAuth } from "../../lib/context/AuthContext";

describe("unit: auth context", () => {
  it("logs in user", async () => {
    const auth = useAuth();
    const credentials = {
      email: "test@example.com",
      password: "password123",
    };

    await auth.login(credentials);

    expect(auth.isAuthenticated).toBe(true);
    expect(auth.user?.email).toBe(credentials.email);
    console.log("User logged in:", auth.user);
  });

  it("logs out user", async () => {
    const auth = useAuth();
    
    await auth.login({ email: "test@example.com", password: "pass" });
    expect(auth.isAuthenticated).toBe(true);

    auth.logout();

    expect(auth.isAuthenticated).toBe(false);
    expect(auth.user).toBeNull();
    console.log("User logged out");
  });

  it("refreshes authentication token", async () => {
    const auth = useAuth();
    
    await auth.login({ email: "test@example.com", password: "pass" });

    await auth.refreshToken();

    expect(auth.isAuthenticated).toBe(true);
    console.log("Token refreshed for user:", auth.user?.email);
  });

  it("updates user profile", async () => {
    const auth = useAuth();
    const newName = "Updated Name";
    
    await auth.login({ email: "test@example.com", password: "pass" });
    const originalName = auth.user?.name;

    auth.updateProfile(newName);

    expect(auth.user?.name).toBe(newName);
    expect(auth.user?.name).not.toBe(originalName);
    console.log("Profile updated from", originalName, "to", newName);
  });
});

describe("unit: auth context workflow", () => {
  it("handles complete authentication flow", async () => {
    const auth = useAuth();

    expect(auth.isAuthenticated).toBe(false);

    await auth.login({
      email: "workflow@example.com",
      password: "test123",
    });
    expect(auth.isAuthenticated).toBe(true);

    await auth.refreshToken();
    expect(auth.isAuthenticated).toBe(true);

    auth.updateProfile("Workflow User");
    expect(auth.user?.name).toBe("Workflow User");

    auth.logout();
    expect(auth.isAuthenticated).toBe(false);

    console.log("Complete workflow executed successfully");
  });

  it("chains multiple profile updates", async () => {
    const auth = useAuth();
    
    await auth.login({ email: "chain@example.com", password: "pass" });

    auth.updateProfile("First Name");
    expect(auth.user?.name).toBe("First Name");

    auth.updateProfile("Second Name");
    expect(auth.user?.name).toBe("Second Name");

    auth.updateProfile("Final Name");
    expect(auth.user?.name).toBe("Final Name");

    console.log("Chained profile updates successful");
  });
});
