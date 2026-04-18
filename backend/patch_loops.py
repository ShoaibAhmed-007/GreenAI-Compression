import re

with open('compress.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix the syntax error from previous run
text = text.replace("print(f\\'      batch {batch_idx+1}/100 loss={loss.item():.4f}\\')", "print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')")

# We want to replace all long-running train_loader loops with batched versions.
# Loop type 1: To device
loop_1 = r"""        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(pruned_model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss \+= loss.item()"""

repl_1 = """        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(pruned_model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')"""
text = re.sub(loop_1, repl_1, text)

# Loop type 2: KD
loop_2 = r"""        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = _extract_logits(student(inputs))
            with torch.no_grad():
                t_out = _extract_logits(teacher(inputs))
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            running_loss \+= loss.item()"""

repl_2 = """        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = _extract_logits(student(inputs))
            with torch.no_grad():
                t_out = _extract_logits(teacher(inputs))
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')"""
text = re.sub(loop_2, repl_2, text)

# Loop type 3: KD ultra
loop_3 = r"""        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = ultra_model(inputs)
            with torch.no_grad():
                t_out = _extract_logits(teacher(inputs))
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            # Re-apply pruning mask to keep zeros at zero
            with torch.no_grad():
                for name, param in ultra_model.named_parameters():
                    if name in pruning_masks:
                        param.data.mul_\(pruning_masks\[name\].to\(param.device\)\)
            running_loss \+= loss.item()"""

repl_3 = """        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_out = ultra_model(inputs)
            with torch.no_grad():
                t_out = _extract_logits(teacher(inputs))
            loss = distillation_loss(s_out, t_out, labels, T=4.0, alpha=0.3)
            loss.backward()
            optimizer.step()
            # Re-apply pruning mask to keep zeros at zero
            with torch.no_grad():
                for name, param in ultra_model.named_parameters():
                    if name in pruning_masks:
                        param.data.mul_(pruning_masks[name].to(param.device))
            running_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')"""

text = re.sub(loop_3, repl_3, text)

# Loop type 4: QAT
loop_4 = r"""                for inputs, labels in train_loader:
                    inputs, labels = inputs.to\(qat_dev\), labels.to\(qat_dev\)

                    optimizer.zero_grad\(\)
                    loss = F.cross_entropy\(_extract_logits\(model\(inputs\)\), labels\)
                    loss.backward\(\)
                    optimizer.step\(\)"""

repl_4 = """                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    if batch_idx >= 100:
                        break
                    inputs, labels = inputs.to(qat_dev), labels.to(qat_dev)

                    optimizer.zero_grad()
                    loss = F.cross_entropy(_extract_logits(model(inputs)), labels)
                    loss.backward()
                    optimizer.step()
                    if (batch_idx + 1) % 20 == 0:
                        print(f'      batch {batch_idx+1}/100 loss={loss.item():.4f}')"""

text = re.sub(loop_4, repl_4, text)

with open('compress.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Patching complete.")
