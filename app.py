from flask import Flask, request, render_template
from main import predict_charges, Convert_USD_to_Ruppee, mean, std, theta

app = Flask(__name__, template_folder='./templates', static_folder='./static')


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    features = [x for x in request.form.values()]
    age = int(features[0])
    sex = int(features[1])
    bmi = float(features[2])
    children = int(features[3])
    smoker = int(features[4])
    region = int(features[5])
    # final = np.array(features).reshape((1, 6))
    # print(final)
    pred = predict_charges(age,sex,bmi,children,smoker,region,mean,std,theta)
    # print(pred)

    if pred < 0:
        return render_template('op.html', pred='Error calculating Amount!')
    else:
        return render_template('op.html', pred=f'USD {round(pred, 2)}$  INR {Convert_USD_to_Ruppee(round(pred, 1))} rs')




if __name__ == '__main__':
    app.run(debug=True)
