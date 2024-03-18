import argparse
import pool_angel

parser = argparse.ArgumentParser(description='Pool Angel')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')

parser.add_argument('--output', '-o', type=str,
                    help='Usage: specify the save path')

parser.add_argument('--det', type=str,default='nanodet',
                    help='Select the detection model')

parser.add_argument('--pose', type=str,default='movenet',
                    help='Select the pose model')

parser.add_argument('--url', type=str,default='https://aadf-106-51-161-33.ngrok-free.app/',
                    help='pool detection url')


args = parser.parse_args()


if __name__ == '__main__':
    pg=pool_angel.PoolAngel(args.input,visualize=args.vis,detector=args.det, posenet=args.pose, save=args.output, server_url=args.url)
    pg.run()


