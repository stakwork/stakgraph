module Config
  PORT = ENV.fetch('PORT', 4567).to_i
  ENV_MODE = ENV.fetch('RACK_ENV', 'development')
  DB_URL = ENV.fetch('DATABASE_URL', 'sqlite3::memory:')
  SECRET_KEY = ENV.fetch('SECRET_KEY', 'change_me_in_production')
end
