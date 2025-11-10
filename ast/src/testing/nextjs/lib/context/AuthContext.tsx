"use client";
import { createContext, useContext, useState, ReactNode } from "react";

interface User {
  id: string;
  email: string;
  name: string;
}

interface Credentials {
  email: string;
  password: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (credentials: Credentials) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  updateProfile: (name: string) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);

  const login = async (credentials: Credentials) => {
    await new Promise((resolve) => setTimeout(resolve, 100));
    
    const mockUser: User = {
      id: "user-123",
      email: credentials.email,
      name: "Test User",
    };
    
    setUser(mockUser);
    setToken("mock-token-abc123");
    console.log("User logged in:", mockUser);
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    console.log("User logged out");
  };

  const refreshToken = async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
    
    if (token) {
      const newToken = `${token}-refreshed-${Date.now()}`;
      setToken(newToken);
      console.log("Token refreshed:", newToken);
    }
  };

  const updateProfile = (name: string) => {
    if (user) {
      setUser({ ...user, name });
      console.log("Profile updated:", name);
    }
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    login,
    logout,
    refreshToken,
    updateProfile,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
