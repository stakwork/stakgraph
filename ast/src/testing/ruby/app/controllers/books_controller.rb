class BooksController < ApplicationController
  # @ast node: Function "index"
  def index
    author = Author.find(params[:author_id])
    books = author.books
    render json: books, status: :ok
  end

  # @ast node: Function "create"
  def create
    author = Author.find(params[:author_id])
    book = author.books.build(book_params)
    if book.save
      render json: book, status: :created
    else
      render json: book.errors, status: :unprocessable_entity
    end
  end

  # @ast node: Function "show"
  def show
    book = Book.find(params[:id])
    render json: book, status: :ok
  end

  private

  # @ast node: Function "book_params"
  def book_params
    params.require(:book).permit(:title, :description, :published_at)
  end
end
