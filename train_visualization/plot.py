import matplotlib.pyplot as plt
import csv
import os

def read_and_preprocess_data(filename):
    # Lê arquivo e salva csv como dicionário
    res = {}
    if os.path.exists(filename):
        with open(filename) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                step = int(row['Step'])
                loss = float(row['Value'])

                if step not in res:
                    res[step] = (1, loss)
                else:
                    counter = res[step][0] + 1
                    sum_loss = res[step][1] + loss
                    res[step] = (counter, sum_loss)
            
            for step in res:
                counter = res[step][0]
                sum_loss = res[step][1]
                res[step] = sum_loss / counter
            
            return res

    else:
        raise Exception('Arquivo %s não existe' % filename)


def plot_loss(data, loss):
    def _plot_loss(d, label):        
        steps, losses = [], []
        for (step, loss) in d.items():
            steps.append(step)
            losses.append(loss)
        
        plt.plot(steps, losses, label=label)

    _plot_loss(data[0], 'Treino')
    _plot_loss(data[1], 'Validação')

    leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.title(loss)
    plt.show()

def main():
    base_filename = 'run_%s-tag-%s.csv'

    for loss in ['heatmaps_mse', 'radius_mse']:
        plot_data = []
        for dataset in ['train', 'test']:
            plot_data.append(read_and_preprocess_data(base_filename % (dataset, loss)))

        plot_loss(plot_data, loss)


if __name__ == '__main__':
    main()
