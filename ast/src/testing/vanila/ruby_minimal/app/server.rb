require 'webrick'
require_relative '../config'
require_relative 'routes'

module Server
  def self.start
    server = WEBrick::HTTPServer.new(Port: Config::PORT)

    server.mount_proc '/' do |req, res|
      status, headers, body = Routes.handle(req)
      
      res.status = status
      headers.each { |k, v| res[k] = v }
      res.body = body.first
    end

    trap 'INT' do server.shutdown end
    
    puts "Starting server on port #{Config::PORT}..."
    server.start
  end
end
