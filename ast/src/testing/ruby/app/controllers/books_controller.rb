class BooksController < ApplicationController
  def index
    author = Author.find(params[:author_id])
    books = author.books
    render json: books, status: :ok
  end

  def create
    author = Author.find(params[:author_id])
    book = author.books.build(book_params)
    if book.save
      render json: book, status: :created
    else
      render json: book.errors, status: :unprocessable_entity
    end
  end

  def show
    book = Book.find(params[:id])
    render json: book, status: :ok
  end

  private

  def book_params
    params.require(:book).permit(:title, :description, :published_at)
  end
end
