# @ast node: Endpoint "/" [verb=GET]
# @ast edge: Handler -> Function "index" "home_controller.rb"
# @ast node: Endpoint "/api/v1/tokens" [verb=POST]
# @ast edge: Handler -> Function "create" "api/v1/tokens_controller.rb"
# @ast node: Endpoint "/api/v1/tokens/:id" [verb=DELETE]
# @ast edge: Handler -> Function "destroy" "api/v1/tokens_controller.rb"
# @ast node: Endpoint "/api/v1/status" [verb=GET]
# @ast edge: Handler -> Function "status" "api/v1/health_controller.rb"
# @ast node: Endpoint "/admin/settings" [verb=GET]
# @ast edge: Handler -> Function "index" "admin/settings_controller.rb"
# @ast node: Endpoint "/admin/settings/:id" [verb=PUT]
# @ast edge: Handler -> Function "update" "admin/settings_controller.rb"
# @ast node: Endpoint "/dashboard" [verb=GET]
# @ast edge: Handler -> Function "show" "dashboards_controller.rb"
# @ast node: Endpoint "/dashboard" [verb=PUT]
# @ast edge: Handler -> Function "update" "dashboards_controller.rb"
# @ast node: Endpoint "/profile" [verb=GET]
# @ast edge: Handler -> Function "show_person_profile" "people_controller.rb"
# @ast node: Endpoint "/profile/edit" [verb=GET]
# @ast edge: Handler -> Function "edit" "profiles_controller.rb"
# @ast node: Endpoint "/profile" [verb=PUT]
# @ast edge: Handler -> Function "update" "profiles_controller.rb"
# @ast node: Endpoint "/authors" [verb=GET]
# @ast edge: Handler -> Function "index" "authors_controller.rb"
# @ast node: Endpoint "/authors" [verb=POST]
# @ast edge: Handler -> Function "create" "authors_controller.rb"
# @ast node: Endpoint "/authors/:id" [verb=GET]
# @ast edge: Handler -> Function "show" "authors_controller.rb"
# @ast node: Endpoint "/authors/:author_id/books" [verb=GET]
# @ast edge: Handler -> Function "index" "books_controller.rb"
# @ast node: Endpoint "/authors/:author_id/books" [verb=POST]
# @ast edge: Handler -> Function "create" "books_controller.rb"
# @ast node: Endpoint "/authors/:author_id/books/:id" [verb=GET]
# @ast edge: Handler -> Function "show" "books_controller.rb"
# @ast node: Endpoint "/person/:id" [verb=GET]
# @ast edge: Handler -> Function "get_person" "people_controller.rb"
# @ast node: Endpoint "/person" [verb=POST]
# @ast edge: Handler -> Function "create_person" "people_controller.rb"
# @ast node: Endpoint "/people/:id" [verb=DELETE]
# @ast edge: Handler -> Function "destroy" "people_controller.rb"
# @ast node: Endpoint "/people/articles" [verb=GET]
# @ast edge: Handler -> Function "articles" "people_controller.rb"
# @ast node: Endpoint "/people/:id/articles" [verb=POST]
# @ast edge: Handler -> Function "articles" "people_controller.rb"
# @ast node: Endpoint "/countries/:country_id/process" [verb=POST]
# @ast edge: Handler -> Function "process" "countries_controller.rb"
Rails.application.routes.draw do
  root to: 'home#index'
  
  namespace :api do
    namespace :v1 do
      resources :tokens, only: [:create, :destroy]
      # Check API status
      get 'status', to: 'health#status'
    end
  end
  
  scope '/admin' do
    resources :settings, only: [:index, :update]
  end
  
  resource :dashboard, only: [:show, :update]
  resource :profile, only: [:show, :edit, :update]
  
  resources :authors do
    resources :books, only: [:index, :create, :show]
  end
  
  get 'person/:id', to: 'people#get_person'
  post 'person', to: 'people#create_person'
  resources :people, only: [:destroy]
  resources :people do
    collection do
      get 'articles'
    end
  end
  resources :people do
    get 'profile', to: 'people#show_person_profile'
    member do
      post 'articles'
    end
  end
  resources :countries do
    post :process
  end
  
end