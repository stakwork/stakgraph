require_relative '../../lib/models'
require_relative '../../lib/utils'

class PostController
  def self.index
    posts = Models::Post.all.map(&:to_h)
    Utils.json_response(posts)
  end

  def self.create(params)
    post = Models::Post.new(params)
    if post.save
      Utils.json_response(post.to_h, 201)
    else
      Utils.json_response({ error: 'Invalid post data' }, 422)
    end
  end
end
