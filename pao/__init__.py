#
# Importing PAO initializes the Pyomo environment, if it
# is necessary, and then registers the pao plugins.
#
# NOTE:  We assume that a user will never import symbols from
# pao directly:
#
#   from pao import *
#
# Instead, users should import symbols directly from pao
# sub-packages:
#
#   from pao.bilevel import *
#
import pyomo.environ
import pao.duality.plugins
import pao.bilevel.plugins
pao.bilevel.plugins.load()

