import argparse
import pool_angel

parser = argparse.ArgumentParser(description='Pool Angel')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
args = parser.parse_args()

if __name__ == '__main__':
    pg=pool_angel.PoolAngel(args.input)
    pg.run()


