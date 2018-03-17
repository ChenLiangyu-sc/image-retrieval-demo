import argparse
import retrieve_demo_gpu
import sys

def main(args):
    retrieve_demo=retrieve_demo_gpu.image_retrieval(args = args)
    retrieve_demo.retrieval(path = args.retrieve_image_dir)
    for dir in range(1000):
        try:
            print('Enter the picture path')
            message = sys.stdin.readline().strip('\n')
            print(retrieve_demo.retrieval(path = message))
        except Exception,e:
            print('Not the path')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str,
    help='Load a pretrained model before training starts.',default='./model1.55293.ckpt-18006')

    parser.add_argument('--initialize_image_dir', type=str,
    help='initialization.', default='/data1/chenliangyu/data/33/201330000071/200330588058.jpg')

    parser.add_argument('--retrieve_image_dir', type=str,
    help='retrieve image', default='/data1/chenliangyu/data/33/201330000071/200330588058.jpg')

    parser.add_argument('--image_mean_dir', type=str,
    help='Image mean', default='./mean_227.npy')



    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
