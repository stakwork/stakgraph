require_relative 'controllers/user_controller'
require_relative 'controllers/post_controller'
require_relative '../lib/utils'

module Routes
  def self.handle(request)
    path = request.path
    method = request.request_method
    body = JSON.parse(request.body.read, symbolize_names: true) rescue {}

    case [method, path]
    when ['GET', '/']
      Utils.json_response({ message: 'Welcome to Ruby Minimal API' })
    when ['GET', '/health']
      Utils.json_response({ status: 'ok' })
    when ['GET', '/users']
      UserController.index
    when ['POST', '/users']
      UserController.create(body)
    when ['GET', %r{^/users/(\d+)$}]
      id = path.scan(%r{^/users/(\d+)$}).flatten.first
      UserController.show(id)
    when ['GET', '/posts']
      PostController.index
    when ['POST', '/posts']
      PostController.create(body)
    else
      Utils.json_response({ error: 'Route not found' }, 404)
    end
  rescue StandardError => e
    Utils.json_response({ error: e.message }, 500)
  end
end
