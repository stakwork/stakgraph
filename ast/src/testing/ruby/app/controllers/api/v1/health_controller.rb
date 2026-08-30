module Api
  module V1
    class HealthController < ApplicationController
      # @ast node: Function "status"
      def status
        render json: { status: 'ok', version: 'v1' }, status: :ok
      end
    end
  end
end
