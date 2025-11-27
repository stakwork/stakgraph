module Api
  module V1
    class TokensController < ApplicationController
      def create
        token = generate_token
        render json: { token: token }, status: :created
      end

      def destroy
        revoke_token(params[:id])
        render json: { message: 'Token revoked' }, status: :ok
      end

      private

      def generate_token
        SecureRandom.hex(32)
      end

      def revoke_token(token_id)
        # Token revocation logic
      end
    end
  end
end
