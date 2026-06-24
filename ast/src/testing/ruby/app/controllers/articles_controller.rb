class ArticlesController < ApplicationController
    private
    # @ast node: Function "article_params"
    def article_params
        params.require(:article).permit(:title, :body)
      end
end