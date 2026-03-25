from flask import Flask,request,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from decouple import config
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#### CONFIGURACION DE SQLALCHEMY ####
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Insurance(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    age = db.Column(db.Integer,nullable=False)
    charges = db.Column(db.Double,nullable=True)
    
    def __init__(self,age):
        self.age = age
        
### CREAMOS UN ESQUEMA PAARA SERIALIZAR LOS DATOS
ma = Marshmallow(app)
class InsuranceSchema(ma.Schema):
    id = ma.Integer()
    age = ma.Integer()
    charges = ma.Float()
    
## REGISTRAMOS LA TABLA EN LA BASE DE DATOS
db.create_all()
print('Tablas en base de datos creadas')


##### INSURANCE ML ################
import joblib
import numpy as np
import sklearn

model = joblib.load('./model/model.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

def predict_charges(age):
    age_sc = sc_x.transform(np.array([[age]]))
    prediction = model.predict(age_sc)
    prediction_sc = sc_y.inverse_transform(prediction)
    charges = round(float(prediction_sc[0][0]),2)
    return charges



@app.route('/')
def index():
    context = {
        'title':'FLASK API VERSION 1.0',
        'message':'AUTOR: Martin Chavez Iglesias'
    }
    return jsonify(context)

@app.route('/insurance_charges',methods=['POST'])
def insurance_charges():
    age = request.json['age']
    charges = predict_charges(age)
    context = {
        'message':'precio predicho',
        'edad': age,
        'charges': charges
    }
    
    return jsonify(context)

###### RUTAS PARA INSURANCE API
@app.route('/insurance',methods=['POST'])
def set_data():
    age = request.json['age']
    charges = predict_charges(age)
    
    #registramos los datos en la tabla
    new_insurance = Insurance(age)
    new_insurance.charges = charges
    db.session.add(new_insurance)
    db.session.commit()
    
    data_schema = InsuranceSchema()
    
    context = data_schema.dump(new_insurance)
    
    return jsonify(context)

@app.route('/insurance',methods=['GET'])
def get_data():
    data = Insurance.query.all() # select * from insurance
    data_schema = InsuranceSchema(many=True)
    return jsonify(data_schema.dump(data))

@app.route('/insurance/<int:id>',methods=['GET'])
def get_data_by_id(id):
    data = Insurance.query.get(id) # select * from insurance where id = id
    data_schema = InsuranceSchema()
    
    return jsonify(data_schema.dump(data)),200 if data else 404

@app.route('/insurance/<int:id>',methods=['PUT'])
def update_data(id):
    data = Insurance.query.get(id) #select * from insurance where id = id
    if not data:
        context = {
            'message':'Registro no encontrado'
        }
        return jsonify(context),404
    
    age = request.json['age']
    charges = predict_charges(age)
    
    data.age = age
    data.charges = charges
    db.session.commit()
    
    data_schema = InsuranceSchema()
    
    return jsonify(data_schema.dump(data)),200

@app.route('/insurance/<int:id>',methods=['DELETE'])
def delete_data(id):
    data = Insurance.query.get(id)
    
    if not data:
        context = {
            'message':'Registro no encontrado'
        }
        return jsonify(context),404
    
    db.session.delete(data) #delete from insurance
    db.session.commit()
    
    context = {
        'message':'Registro eliminado correctamente'
    }
    
    return jsonify(context),200

    
if __name__ == '__main__':
    app.run(debug=True)