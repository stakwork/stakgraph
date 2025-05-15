Rails.application.routes.draw do
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