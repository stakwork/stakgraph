Rails.application.routes.draw do
  root to: 'home#index'
  
  namespace :api do
    namespace :v1 do
      resources :tokens, only: [:create, :destroy]
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