# agentsLister
import os

from optparse import OptionParser

usageStr = "-s short mode"
parser = OptionParser(usageStr)
parser.add_option('-s', '--short', action='store_true', dest='short',
                  help='Generate minimal output and no graphics', default=False)

options, otherjunk = parser.parse_args()


# Looks through all pythonPath Directories for the right module,
pythonPathStr = os.path.expandvars("$PYTHONPATH")
if pythonPathStr.find(';') == -1:
    pythonPathDirs = pythonPathStr.split(':')
else:
    pythonPathDirs = pythonPathStr.split(';')
#pythonPathDirs.append('.')

pythonPathDirs=[]
pythonPathDirs.append('.')

for moduleDir in pythonPathDirs:
    if not options.short:
        print 'Dirs' + str(pythonPathDirs)
        print

    if os.path.isdir(moduleDir):
        moduleNames = [f for f in os.listdir(moduleDir) if f.endswith('_Agents.py')]

        if not options.short:
            print 'Modules ' + str(moduleNames)
            print

        for modulename in moduleNames:
            name = modulename[:-10]
            module = __import__(modulename[:-3])
            if 'CREATORS' in dir(module):
                if options.short:
                    print name
                else:
                    creators = module.CREATORS
                    print('Module=' + modulename)
                    print('  AgentName=' + name)
                    for creator in enumerate(creators):
                        i = creator[0]
                        id = creator[1]
                        print('    Creator ' + str(i + 1) + ' ' + str(id) + '>' + creators[id])
                    print "\n"
