export const CONFIG = {
  PORT: process.env.PORT || 3000,
  DB_URI: process.env.DB_URI || "memory://db",
  NODE_ENV: process.env.NODE_ENV || "development",
  SECRET: "development_secret_key",
};
