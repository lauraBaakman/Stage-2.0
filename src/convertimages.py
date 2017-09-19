import argparse
import logging
import subprocess

from unipath import Path

from argparseActions import InputDirectoryAction, OutputDirectoryAction
import inputoutput.utils as ioUtils

_default_input_path = Path('../paper')
_default_exception_list = [Path('../paper/paper.pdf')]

args = None

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--image-directory',
                        action=InputDirectoryAction,
                        default=_default_input_path,
                        help='''The folder to recursively search for images to be converted.''')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="Increase the output verbosity.")
    parser.add_argument('-r', '--replace-all-existing',
                        action='store_true',
                        default=False,
                        help="Overwrite all existing png files.")
    parser.add_argument('-n', '--replace-newer',
                        action='store_true',
                        default=True,
                        help="Overwrite existing png files if the pdf file is newer.")
    parser.add_argument('--dry-run',
                        action='store_true',
                        default=False,
                        help="Perform a dry run, it is helpful to increase the output verbosity as well.")
    parser.add_argument('-e','--files-to-skip',
                        nargs='*',
                        help='PDF files that should not be converted',
                        default=_default_exception_list,
                        required=False)
    return parser


def find_pdf_files(directory):
    def is_pdf_file(file):
        return file.ext == Path(u'.pdf')

    def is_not_in_exception_list(file):
        return not(file in args.files_to_skip)

    file_filter = lambda x : is_pdf_file(x) and is_not_in_exception_list(x)
    return list(directory.walk(filter=file_filter))


def convert_to_absolute_paths(files):
    files = [file.absolute() for file in files]
    return files


def convert_pdf_file(file):
    def build_png_path(file):
        return file.parent.child('{}.png'.format(file.stem))

    png_path = build_png_path(file)

    if not(args.replace_all_existing) and args.replace_newer and file.ctime() < png_path.ctime():
        logging.info('An up-to-date png file alread exists for ...{pdf_file}'.format(
            pdf_file=ioUtils.partial_path(file))
        )
        return

    if not(args.replace_all_existing) and not(args.replace_newer) and png_path.exists():
        logging.info('A png file alread exists for ...{pdf_file}'.format(
            pdf_file=ioUtils.partial_path(file))
        )
        return

    logging.info(
        'Converting ...{pdf_file} to ...{png_file}.'.format(
            pdf_file=ioUtils.partial_path(file),
            png_file=ioUtils.partial_path(png_path)
        )
    )

    if  args.dry_run:
        return

    # convert -density 150 -antialias ~/Desktop/overlay.pdf -quality 100 ~/Desktop/temp.png
    cli_args = [
        'convert', '-density', '150', '-antialias', file,
        '-quality', '100', png_path
    ]
    subprocess.call(cli_args)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.ERROR)
    if args.dry_run:
        logging.info('This is dry run!')
    logging.info(
        'Skipping:\n\t{}'.format(
            '\n\t'.join([ioUtils.partial_path(file) for file in args.files_to_skip])
        )
    )

    files = find_pdf_files(args.image_directory)
    files = convert_to_absolute_paths(files)
    for file in files:
        convert_pdf_file(file)
