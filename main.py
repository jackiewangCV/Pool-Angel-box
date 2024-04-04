import argparse
import pool_angel

parser = argparse.ArgumentParser(description='Pool Angel')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')

parser.add_argument('--url', type=str,default='local',
                    help='pool detection url')


args = parser.parse_args()


if __name__ == '__main__':
    pg=pool_angel.PoolAngel(args.input, server_url=args.url)
    pg.run()


