# # Loss
# parser.add_argument(
#     "--label_smoothing", default=0.1, type=float, help="label smoothing"
# )
#
# # Training config
# parser.add_argument(
#     "--epochs", default=150, type=int, help="Number of maximum epochs"
# )
# # minibatch
# parser.add_argument(
#     "--shuffle", default=1, type=int, help="reshuffle the data at every epoch"
# )
# parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
# parser.add_argument(
#     "--batch_frames",
#     default=0,
#     type=int,
#     help="Batch frames. If this is not 0, batch size will make no sense",
# )
# parser.add_argument(
#     "--maxlen-in",
#     default=800,
#     type=int,
#     metavar="ML",
#     help="Batch size is reduced if the input sequence length > ML",
# )
# parser.add_argument(
#     "--maxlen-out",
#     default=150,
#     type=int,
#     metavar="ML",
#     help="Batch size is reduced if the output sequence length > ML",
# )
# parser.add_argument(
#     "--num-workers",
#     default=4,
#     type=int,
#     help="Number of workers to generate minibatch",
# )
# # optimizer
# parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
# parser.add_argument(
#     "--k", default=0.2, type=float, help="tunable scalar multiply to learning rate"
# )
# parser.add_argument("--warmup_steps", default=4000, type=int, help="warmup steps")
#
# parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")
