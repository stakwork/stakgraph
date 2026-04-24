import express from 'express';
import { authenticate } from '../middleware/auth';
import { PostService } from '../services/post-service';

const router = express.Router({ mergeParams: true }); // Important for accessing parent params
const postService = new PostService();

// Get all posts for a user
router.get('/', async (req, res) => {
  const userId = req.params.userId; // From parent route
  const posts = await postService.findByUserId(userId);
  res.json(posts);
});

// Get specific post
router.get('/:postId', async (req, res) => {
  const { userId, postId } = req.params;
  const post = await postService.findOne(userId, postId);
  res.json(post);
});

// Create post for user
router.post('/', authenticate, async (req, res) => {
  const userId = req.params.userId;
  const post = await postService.create(userId, req.body);
  res.status(201).json(post);
});

// Method chaining pattern
router.route('/:postId/like')
  .post(async (req, res) => {
    await postService.likePost(req.params.postId);
    res.status(200).end();
  })
  .delete(async (req, res) => {
    await postService.unlikePost(req.params.postId);
    res.status(200).end();
  });

export default router;
// @ast node: Function "get_handler_L8"
// @ast node: Function "get_param_postId_handler_L15"
// @ast node: Function "post_handler_L22"
// @ast node: Function "post_param_postId_like_handler_L30"
// @ast node: Function "delete_param_postId_like_handler_L34"
// @ast node: Endpoint "/:userId/posts/" [verb=GET]
// @ast node: Endpoint "/:userId/posts/" [verb=POST]
// @ast node: Endpoint "/:userId/posts/:postId" [verb=GET]
// @ast node: Endpoint "/:userId/posts/:postId/like" [verb=POST]
// @ast node: Endpoint "/:userId/posts/:postId/like" [verb=DELETE]
// @ast node: Var "postService"