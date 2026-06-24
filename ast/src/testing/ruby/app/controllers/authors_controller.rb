class AuthorsController < ApplicationController
  # @ast node: Function "index"
  def index
    authors = Author.all
    render json: authors, status: :ok
  end

  # @ast node: Function "create"
  def create
    author = Author.new(author_params)
    if author.save
      render json: author, status: :created
    else
      render json: author.errors, status: :unprocessable_entity
    end
  end

  # @ast node: Function "show"
  def show
    author = Author.find(params[:id])
    render json: author, status: :ok
  end

  private

  # @ast node: Function "author_params"
  def author_params
    params.require(:author).permit(:name, :bio)
  end
end
