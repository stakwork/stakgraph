require 'singleton'

class Database
  include Singleton

  def initialize
    @store = {
      users: {},
      posts: {}
    }
  end

  def save(collection, id, record)
    @store[collection][id] = record
  end

  def find(collection, id)
    @store[collection][id]
  end

  def all(collection)
    @store[collection].values
  end

  def delete(collection, id)
    @store[collection].delete(id)
  end

  def clear!
    @store.each_value(&:clear)
  end
end

DB = Database.instance
