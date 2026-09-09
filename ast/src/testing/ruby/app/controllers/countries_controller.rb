class CountriesController < ApplicationController
    # @ast node: Function "process"
    def process
        country = Country.new(country_params)
        if country.save
            render json: country, status: :created
          else
            render json: country.errors, status: :unprocessable_entity
          end
        end
    end

    private

    # @ast node: Function "country_params"
    def country_params
        params.require(:country).permit(:name, :code)
    end
end