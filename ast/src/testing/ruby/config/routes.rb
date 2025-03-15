Rails.application.routes.draw do
  get 'person/:id', to: 'people#get_person'
  post 'person', to: 'people#create_person'
end