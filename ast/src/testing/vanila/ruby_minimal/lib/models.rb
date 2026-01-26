require_relative '../database'
require_relative 'utils'
require_relative 'validators'

module Models
  class Base
    attr_accessor :id, :created_at

    def initialize(attributes = {})
      @id = attributes[:id] || Utils.generate_id
      @created_at = attributes[:created_at] || Time.now
    end

    def save
      return false unless valid?
      DB.save(collection_name, @id, self)
      true
    end

    def self.find(id)
      DB.find(collection_name, id)
    end

    def self.all
      DB.all(collection_name)
    end
  end

  class User < Base
    attr_accessor :username, :email

    def initialize(attributes = {})
      super
      @username = attributes[:username]
      @email = attributes[:email]
    end

    def self.collection_name
      :users
    end

    def collection_name
      :users
    end

    def valid?
      Validators.present?(@username) && Validators.valid_email?(@email)
    end

    def to_h
      { id: @id, username: @username, email: @email, created_at: @created_at }
    end
  end

  class Post < Base
    attr_accessor :user_id, :title, :content

    def initialize(attributes = {})
      super
      @user_id = attributes[:user_id]
      @title = attributes[:title]
      @content = attributes[:content]
    end

    def self.collection_name
      :posts
    end

    def collection_name
      :posts
    end

    def valid?
      Validators.present?(@title) && Validators.present?(@content) && !@user_id.nil?
    end

    def to_h
      { id: @id, user_id: @user_id, title: @title, content: @content, created_at: @created_at }
    end
  end
end
