import React, {
  forwardRef,
  memo,
  useContext,
  createContext,
  ComponentType,
} from "react";

// Context for authentication
interface AuthContextType {
  isAuthenticated: boolean;
  user: string | null;
}

export const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  user: null,
});

// Higher-Order Component for authentication
// @ast node: Function "withAuth"
export function withAuth<P extends object>(WrappedComponent: ComponentType<P>) {
  return function WithAuthComponent(props: P) {
    const auth = useContext(AuthContext);

    if (!auth.isAuthenticated) {
      return <div>Please log in</div>;
    }

    return <WrappedComponent {...props} />;
  };
}

// Higher-Order Component for loading state
// @ast node: Function "withLoading"
export function withLoading<P extends { loading?: boolean }>( 
  WrappedComponent: ComponentType<P>
) {
  return function WithLoadingComponent(props: P) {
    if (props.loading) {
      return <div>Loading...</div>;
    }
    return <WrappedComponent {...props} />;
  };
}

// Memoized component
interface MemoizedCardProps {
  title: string;
  content: string;
}

// @ast node: Function "MemoizedCard"
export const MemoizedCard = memo(function MemoizedCard({
  title,
  content,
}: MemoizedCardProps) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <p>{content}</p>
    </div>
  );
});

// ForwardRef component
interface InputProps {
  label: string;
  placeholder?: string;
}

// @ast node: Function "ForwardedInput"
export const ForwardedInput = forwardRef<HTMLInputElement, InputProps>(
  function ForwardedInput({ label, placeholder }, ref) {
    return (
      <div>
        <label>{label}</label>
        <input ref={ref} placeholder={placeholder} />
      </div>
    );
  }
);

// Auth Provider component
interface AuthProviderProps {
  children: React.ReactNode;
}

// @ast node: Function "AuthProvider"
export function AuthProvider({ children }: AuthProviderProps) {
  const value: AuthContextType = {
    isAuthenticated: true,
    user: "testuser",
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Hook using the context
// @ast node: Function "useAuth"
export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
// @ast node: Function "AuthContext"
// @ast node: DataModel "AuthContextType"
// @ast node: DataModel "MemoizedCardProps"
// @ast node: DataModel "InputProps"
// @ast node: DataModel "AuthProviderProps"
