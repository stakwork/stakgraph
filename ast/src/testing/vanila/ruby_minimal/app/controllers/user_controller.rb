require_relative '../../lib/models'
require_relative '../../lib/utils'

class UserController
  def self.index
    users = Models::User.all.map(&:to_h)
    Utils.json_response(users)
  end

  def self.show(id)
    user = Models::User.find(id)
    if user
      Utils.json_response(user.to_h)
    else
      Utils.json_response({ error: 'User not found' }, 404)
    end
  end

  def self.create(params)
    user = Models::User.new(params)
    if user.save
      Utils.json_response(user.to_h, 201)
    else
      Utils.json_response({ error: 'Invalid user data' }, 422)
    end
  end
end
