# To be used in conjunction with https://purple-mud-05cbf810f.2.azurestaticapps.net/
# or any other deployment of CDE's web database viewer, where you can paste in the URL
# displayed when you run this script

import e2e_workflow.labelling.server as database_viewer
from photocatalyst_models import ApparentQuantumYield, SolarToHydrogen, HydrogenEvolution, HydrogenEvolution2, HydrogenEvolution3

database_viewer.MODELS = [ApparentQuantumYield, SolarToHydrogen, HydrogenEvolution, HydrogenEvolution2, HydrogenEvolution3]

database_viewer.app.run(debug=True, port=5001)
