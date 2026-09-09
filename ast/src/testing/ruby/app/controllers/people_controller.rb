# @ast node: Var "MAX_PEOPLE"
# Max people limit
MAX_PEOPLE = 100

class PeopleController < ApplicationController
  # Retrieves a person by ID
  # @ast node: Function "get_person"
  def get_person
    person = PersonService.get_person_by_id(params[:id])

    if person
      render json: person, status: :ok
    else
      render json: { error: "Person not found" }, status: :not_found
    end
  end

  # @ast node: Function "create_person"
  def create_person
    person = PersonService.new_person(person_params)

    if person.persisted?
      render json: person, status: :created
    else
      render json: { errors: person.errors.full_messages }, status: :unprocessable_entity
    end
  end

  # @ast node: Function "destroy"
  def destroy
    deleted_person = PersonService.delete(params[:id])

    if deleted_person.destroyed?
      render json: { message: 'Person deleted successfully' }, status: :ok
    else
      render json: { error: "Person not found" }, status: :not_found
    end
  end

  # @ast node: Function "articles"
  def articles
    articles = Article.all
    render json: articles, status: :ok
  end

  # @ast node: Function "create_article"
  def create_article
    person = Person.find(params[:id])
    article = person.articles.build(article_params)

    if article.save
      render json: article, status: :created
    else
      render json: article.errors, status: :unprocessable_entity
    end
  end

  # @ast node: Function "show_person_profile"
  def show_person_profile
    @person = Person.find(params[:id])
  end

  private

  # @ast node: Function "person_params"
  def person_params
    params.require(:person).permit(:name, :email)
  end
end