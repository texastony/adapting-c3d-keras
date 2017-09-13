
import hachoir_core
import hachoir_core.cmd_line
import hachoir_metadata
import hachoir_parser
import sys

# TODO: add TrID
# I changed this
# https://bitbucket.org/haypo/hachoir/wiki/hachoir-metadata/code


def getMetaData(filename):
    text = ""
    filename, realname = hachoir_core.cmd_line.unicodeFilename(
        filename), filename
    parser = hachoir_parser.createParser(filename, realname)

    if not parser:
        print >>sys.stderr, "Unable to parse file"
        return text
    try:
        metadata = hachoir_metadata.extractMetadata(parser)
    except HachoirError:
        metadata = None
    if not metadata:
        print >>sys.stderr, "Unable to extract metadata"
        return text
    text = metadata.exportPlaintext()
    # print meta_data_text
    for line in text:
        print hachoir_core.tools.makePrintable(line,
                                               hachoir_core.i18n.getTerminalCharset())


if __name__ == "__main__":
    if 1 < len(sys.argv):
        getMetaData(sys.argv[1])
