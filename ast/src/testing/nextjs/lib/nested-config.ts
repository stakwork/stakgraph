export const authOptions = {
  providers: [
    {
      id: 'credentials',
      name: 'Credentials',
    }
  ],
  callbacks: {
    async signIn({ user, account, profile }) {
      if (!user.email) {
        return false;
      }
      
      const existingUser = await findUserByEmail(user.email);
      if (!existingUser) {
        await createUser(user);
      }
      
      return true;
    },
    
    async jwt({ token, user, account }) {
      if (user) {
        token.userId = user.id;
        token.role = user.role;
      }
      return token;
    },
    
    async session({ session, token }) {
      if (token) {
        session.user.id = token.userId as string;
        session.user.role = token.role as string;
      }
      return session;
    }
  },
  pages: {
    signIn: '/auth/signin',
    error: '/auth/error',
  }
};

export const apiConfig = {
  endpoints: {
    users: {
      list: async () => {
        const response = await fetch('/api/users');
        return response.json();
      },
      create: async (data: any) => {
        const response = await fetch('/api/users', {
          method: 'POST',
          body: JSON.stringify(data),
        });
        return response.json();
      },
    },
    posts: {
      list: async () => {
        const response = await fetch('/api/posts');
        return response.json();
      },
    }
  },
  middleware: {
    auth: async (req: any) => {
      const token = req.headers.authorization;
      if (!token) {
        throw new Error('Unauthorized');
      }
      return true;
    },
    logger: (req: any) => {
      console.log(`${req.method} ${req.url}`);
    }
  }
};

export const routerConfig = {
  routes: {
    home: '/',
    dashboard: {
      path: '/dashboard',
      handler: async () => {
        return { title: 'Dashboard' };
      }
    },
    settings: {
      path: '/settings',
      nested: {
        profile: {
          path: '/settings/profile',
          handler: async () => {
            return { title: 'Profile Settings' };
          }
        }
      }
    }
  }
};

async function findUserByEmail(email: string) {
  return null;
}

async function createUser(user: any) {
  return user;
}
