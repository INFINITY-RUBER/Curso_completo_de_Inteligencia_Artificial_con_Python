

import json


class ParamsManager(object):
    
    def __init__(self, params_file):# Cargamos lo parametros de forma solo lectura donde params_file: es la ruta del json
        self.params = json.load(open(params_file, 'r'))
        
    def get_params(self):
        return self.params

    def get_agent_params(self): # devolvera solo Dicionario parametros del Agente json
        return self.params['agent']

    def get_environment_params(self): # devolvera solo Dicionario parametros del environment json
        return self.params['environment']

    def update_agent_params(self, **kwargs):# metodo para Actulaizar los valores y reccorrer el dicionario del agente
        for key, value in kwargs.items():
            if key in self.get_agent_params().keys():
                self.params['agent'][key] = value

    def export_agent_params(self, file_name): # metodo para crear un json si se modifican los parameteros del agent
        with open(file_name, 'w') as f:  
            json.dump(self.params['agent'], f, indent=4, separators = (',', ':'), sort_keys=True)
            f.write("\n")
                 
    def export_environment_params(self, file_name):
        with open(file_name, 'w') as f:   # metodo para crear un json si se modifican los parameteros del environment
            json.dump(self.params['environment'], f, indent=4, separators = (',', ':'), sort_keys=True)
            f.write("\n")  



if __name__ == "__main__":
    print("Probando nuestro manager de parámetros...")
    param_file = "./GPU_parameters.json"
    manager = ParamsManager(param_file)
    
    agent_params = manager.get_agent_params()
    print("Los parámetros del agente son: ")
    for key, value in agent_params.items():
        print(key, ": ", value)

    env_params = manager.get_environment_params()
    print("Los parámetros del entorno son: ")
    for key, value in env_params.items():
        print(key, ": ", value)
        
        
    manager.update_agent_params(learning_rate = 0.01, gamma = 0.92)
    agent_params_updated = manager.get_agent_params()
    print("Los parámetros del agente actualizados son: ")
    for key, value in agent_params_updated.items():
        print(key, ": ", value)
    
    print("Fin de la prueba")